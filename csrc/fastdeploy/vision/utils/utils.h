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

#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace utils {
// topk sometimes is a very small value
// so this implementation is simple but I don't think it will
// cost too much time
// Also there may be cause problem since we suppose the minimum value is
// -99999999
// Do not use this function on array which topk contains value less than
// -99999999
template <typename T>
std::vector<int32_t> TopKIndices(const T* array, int array_size, int topk) {
  topk = std::min(array_size, topk);
  std::vector<int32_t> res(topk);
  std::set<int32_t> searched;
  for (int32_t i = 0; i < topk; ++i) {
    T min = -99999999;
    for (int32_t j = 0; j < array_size; ++j) {
      if (searched.find(j) != searched.end()) {
        continue;
      }
      if (*(array + j) > min) {
        res[i] = j;
        min = *(array + j);
      }
    }
    searched.insert(res[i]);
  }
  return res;
}

template <typename T>
void ArgmaxScoreMap(T infer_result_buffer, SegmentationResult* result,
                    bool with_softmax) {
  int64_t height = result->shape[0];
  int64_t width = result->shape[1];
  int64_t num_classes = result->shape[2];
  int index = 0;
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      int64_t s = (i * width + j) * num_classes;
      T max_class_score = std::max_element(
          infer_result_buffer + s, infer_result_buffer + s + num_classes);
      int label_id = std::distance(infer_result_buffer + s, max_class_score);
      if (label_id >= 255) {
        FDWARNING << "label_id is stored by uint8_t, now the value is bigger "
                     "than 255, it's "
                  << static_cast<int>(label_id) << "." << std::endl;
      }
      result->label_map[index] = static_cast<uint8_t>(label_id);

      if (with_softmax) {
        double_t total = 0;
        for (int k = 0; k < num_classes; k++) {
          total += exp(*(infer_result_buffer + s + k) - *max_class_score);
        }
        double_t softmax_class_score = 1 / total;
        result->score_map[index] = static_cast<float>(softmax_class_score);

      } else {
        result->score_map[index] = static_cast<float>(*max_class_score);
      }
      index++;
    }
  }
}

template <typename T>
void NCHW2NHWC(FDTensor& infer_result) {
  T* infer_result_buffer = reinterpret_cast<T*>(infer_result.MutableData());
  int num = infer_result.shape[0];
  int channel = infer_result.shape[1];
  int height = infer_result.shape[2];
  int width = infer_result.shape[3];
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  std::vector<T> hwc_data(chw);
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] =
              *(infer_result_buffer + index);
          index++;
        }
      }
    }
  }
  std::memcpy(infer_result.MutableData(), hwc_data.data(),
              num * chw * sizeof(T));
  infer_result.shape = {num, height, width, channel};
}

void NMS(DetectionResult* output, float iou_threshold = 0.5);

void NMS(FaceDetectionResult* result, float iou_threshold = 0.5);

// MergeSort
void SortDetectionResult(DetectionResult* output);

void SortDetectionResult(FaceDetectionResult* result);

// L2 Norm / cosine similarity  (for face recognition, ...)
FASTDEPLOY_DECL std::vector<float> L2Normalize(
    const std::vector<float>& values);

FASTDEPLOY_DECL float CosineSimilarity(const std::vector<float>& a,
                                       const std::vector<float>& b,
                                       bool normalized = true);

void CropImg(cv::Mat& img, cv::Mat& crop_img, std::vector<int>& area,
             std::vector<float>& center, std::vector<float>& scale,
             float expandratio = 0.15);

void dark_parse(std::vector<float>& heatmap, std::vector<int>& dim,
                std::vector<float>& coords, int px, int py, int index, int ch);

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
