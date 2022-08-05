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
enum FASTDEPLOY_DECL ResultType {
  UNKNOWN_RESULT,
  CLASSIFY,
  DETECTION,
  SEGMENTATION,
  FACE_DETECTION
};

struct FASTDEPLOY_DECL BaseResult {
  ResultType type = ResultType::UNKNOWN_RESULT;
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

  std::string Str();
};

struct FASTDEPLOY_DECL FaceDetectionResult : public BaseResult {
  // box: xmin, ymin, xmax, ymax
  std::vector<std::array<float, 4>> boxes;
  // landmark: x, y, landmarks may empty if the
  // model don't detect face with landmarks.
  // Note, one face might have multiple landmarks,
  // such as 5/19/21/68/98/..., etc.
  std::vector<std::array<float, 2>> landmarks;
  std::vector<float> scores;
  ResultType type = ResultType::FACE_DETECTION;
  // set landmarks_per_face manually in your post processes.
  int landmarks_per_face;

  FaceDetectionResult() { landmarks_per_face = 0; }
  FaceDetectionResult(const FaceDetectionResult& res);

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  std::string Str();
};

struct FASTDEPLOY_DECL SegmentationResult : public BaseResult {
  // mask
  std::vector<uint8_t> label_map;
  std::vector<float> score_map;
  std::vector<int64_t> shape;
  bool contain_score_map = false;

  ResultType type = ResultType::SEGMENTATION;

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  std::string Str();
};

}  // namespace vision
}  // namespace fastdeploy
