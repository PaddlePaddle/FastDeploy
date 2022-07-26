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

void NMS(DetectionResult* output, float iou_threshold = 0.5);

void NMS(FaceDetectionResult* result, float iou_threshold = 0.5);

// MergeSort
void SortDetectionResult(DetectionResult* output);

void SortDetectionResult(FaceDetectionResult* result);

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
