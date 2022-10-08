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
/** \brief All C++ FastDeploy Vision Models APIs are defined inside this namespace
*
*/
namespace vision {
enum FASTDEPLOY_DECL ResultType {
  UNKNOWN_RESULT,
  CLASSIFY,
  DETECTION,
  SEGMENTATION,
  OCR,
  FACE_DETECTION,
  FACE_RECOGNITION,
  MATTING,
  MASK
};

struct FASTDEPLOY_DECL BaseResult {
  ResultType type = ResultType::UNKNOWN_RESULT;
};

/*! @brief Classify result structure for all the image classify models
 */
struct FASTDEPLOY_DECL ClassifyResult : public BaseResult {
  /// Classify result for an image
  std::vector<int32_t> label_ids;
  /// The confidence for each classify result
  std::vector<float> scores;
  ResultType type = ResultType::CLASSIFY;

  /// Clear result
  void Clear();

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! Mask structure, used in DetectionResult for instance segmentation models
 */
struct FASTDEPLOY_DECL Mask : public BaseResult {
  /// Mask data buffer
  std::vector<int32_t> data;
  /// Shape of mask
  std::vector<int64_t> shape;  // (H,W) ...
  ResultType type = ResultType::MASK;

  /// clear mask
  void Clear();

  /// Return a mutable pointer of the mask data buffer
  void* Data() { return data.data(); }

  /// Return a pointer of the mask data buffer for read only
  const void* Data() const { return data.data(); }

  /// Reserve size for mask data buffer
  void Reserve(int size);

  /// Resize the mask data buffer
  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};


/*! @brief Detection result structure for all the object detection models and instance segmentation models
 */
struct FASTDEPLOY_DECL DetectionResult : public BaseResult {
  /** \brief All the detected object boxes for an input image, the size of `boxes` is the number of detected objects, and the element of `boxes` is a array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<float, 4>> boxes;
  /** \brief The confidence for all the detected objects
   */
  std::vector<float> scores;
  /// The classify label for all the detected objects
  std::vector<int32_t> label_ids;
  /** \brief For instance segmentation model, `masks` is the predict mask for all the deteced objects
   */
  std::vector<Mask> masks;
  //// Shows if the DetectionResult has mask
  bool contain_masks = false;

  ResultType type = ResultType::DETECTION;

  DetectionResult() {}
  DetectionResult(const DetectionResult& res);

  /// Clear detection result
  void Clear();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

struct FASTDEPLOY_DECL KeyPointDetectionResult : public BaseResult {
  // Keypoints: shape(N x 3); N: number of Joints; 3: x,y,conf
  std::vector<float> keypoints;
  int num_joints = -1;

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  std::string Str();
};

struct FASTDEPLOY_DECL OCRResult : public BaseResult {
  std::vector<std::array<int, 8>> boxes;

  std::vector<std::string> text;
  std::vector<float> rec_scores;

  std::vector<float> cls_scores;
  std::vector<int32_t> cls_labels;

  ResultType type = ResultType::OCR;

  void Clear();

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

struct FASTDEPLOY_DECL FaceRecognitionResult : public BaseResult {
  // face embedding vector with 128/256/512 ... dim
  std::vector<float> embedding;

  ResultType type = ResultType::FACE_RECOGNITION;

  FaceRecognitionResult() {}
  FaceRecognitionResult(const FaceRecognitionResult& res);

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  std::string Str();
};

struct FASTDEPLOY_DECL MattingResult : public BaseResult {
  // alpha matte and fgr (predicted foreground: HWC/BGR float32)
  std::vector<float> alpha;       // h x w
  std::vector<float> foreground;  // h x w x c (c=3 default)
  // height, width, channel for foreground and alpha
  // must be (h,w,c) and setup before Reserve and Resize
  // c is only for foreground if contain_foreground is true.
  std::vector<int64_t> shape;
  bool contain_foreground = false;

  ResultType type = ResultType::MATTING;

  MattingResult() {}
  MattingResult(const MattingResult& res);

  void Clear();

  void Reserve(int size);

  void Resize(int size);

  std::string Str();
};

}  // namespace vision
}  // namespace fastdeploy
