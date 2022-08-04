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

FaceDetectionResult::FaceDetectionResult(const FaceDetectionResult& res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  landmarks.assign(res.landmarks.begin(), res.landmarks.end());
  scores.assign(res.scores.begin(), res.scores.end());
  landmarks_per_face = res.landmarks_per_face;
}

void FaceDetectionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<std::array<float, 2>>().swap(landmarks);
  landmarks_per_face = 0;
}

void FaceDetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  if (landmarks_per_face > 0) {
    landmarks.reserve(size * landmarks_per_face);
  }
}

void FaceDetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  if (landmarks_per_face > 0) {
    landmarks.resize(size * landmarks_per_face);
  }
}

std::string FaceDetectionResult::Str() {
  std::string out;
  // format without landmarks
  if (landmarks_per_face <= 0) {
    out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score]\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
      out = out + std::to_string(boxes[i][0]) + "," +
            std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
            ", " + std::to_string(boxes[i][3]) + ", " +
            std::to_string(scores[i]) + "\n";
    }
    return out;
  }
  // format with landmarks
  FDASSERT((landmarks.size() == boxes.size() * landmarks_per_face),
           "The size of landmarks != boxes.size * landmarks_per_face.");
  out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score, (x, y) x " +
        std::to_string(landmarks_per_face) + "]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", ";
    for (size_t j = 0; j < landmarks_per_face; ++j) {
      out = out + "(" +
            std::to_string(landmarks[i * landmarks_per_face + j][0]) + "," +
            std::to_string(landmarks[i * landmarks_per_face + j][1]);
      if (j < landmarks_per_face - 1) {
        out = out + "), ";
      } else {
        out = out + ")\n";
      }
    }
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
  out = "SegmentationResult Image masks 10 rows x 10 cols: \n";
  for (size_t i = 0; i < 10; ++i) {
    out += "[";
    for (size_t j = 0; j < 10; ++j) {
      out = out + std::to_string(masks[i][j]) + ", ";
    }
    out += ".....]\n";
  }
  out += "...........\n";
  return out;
}

FaceRecognitionResult::FaceRecognitionResult(const FaceRecognitionResult& res) {
  embedding.assign(res.embedding.begin(), res.embedding.end());
}

void FaceRecognitionResult::Clear() { std::vector<float>().swap(embedding); }

void FaceRecognitionResult::Reserve(int size) { embedding.reserve(size); }

void FaceRecognitionResult::Resize(int size) { embedding.resize(size); }

std::string FaceRecognitionResult::Str() {
  std::string out;
  out = "FaceRecognitionResult: [";
  size_t numel = embedding.size();
  if (numel <= 0) {
    return out + "Empty Result]";
  }
  // max, min, mean
  float min_val = embedding.at(0);
  float max_val = embedding.at(0);
  float total_val = embedding.at(0);
  for (size_t i = 1; i < numel; ++i) {
    float val = embedding.at(i);
    total_val += val;
    if (val < min_val) {
      min_val = val;
    }
    if (val > max_val) {
      max_val = val;
    }
  }
  float mean_val = total_val / static_cast<float>(numel);
  out = out + "Dim(" + std::to_string(numel) + "), " + "Min(" +
        std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
        "), " + "Mean(" + std::to_string(mean_val) + ")]\n";
  return out;
}

}  // namespace vision
}  // namespace fastdeploy
