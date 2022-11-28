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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace utils {

void Merge(DetectionResult* result, size_t low, size_t mid, size_t high) {
  std::vector<std::array<float, 4>>& boxes = result->boxes;
  std::vector<float>& scores = result->scores;
  std::vector<int32_t>& label_ids = result->label_ids;
  std::vector<std::array<float, 4>> temp_boxes(boxes);
  std::vector<float> temp_scores(scores);
  std::vector<int32_t> temp_label_ids(label_ids);
  size_t i = low;
  size_t j = mid + 1;
  size_t k = i;
  for (; i <= mid && j <= high; k++) {
    if (temp_scores[i] >= temp_scores[j]) {
      scores[k] = temp_scores[i];
      label_ids[k] = temp_label_ids[i];
      boxes[k] = temp_boxes[i];
      i++;
    } else {
      scores[k] = temp_scores[j];
      label_ids[k] = temp_label_ids[j];
      boxes[k] = temp_boxes[j];
      j++;
    }
  }
  while (i <= mid) {
    scores[k] = temp_scores[i];
    label_ids[k] = temp_label_ids[i];
    boxes[k] = temp_boxes[i];
    k++;
    i++;
  }
  while (j <= high) {
    scores[k] = temp_scores[j];
    label_ids[k] = temp_label_ids[j];
    boxes[k] = temp_boxes[j];
    k++;
    j++;
  }
}

void MergeSort(DetectionResult* result, size_t low, size_t high) {
  if (low < high) {
    size_t mid = (high - low) / 2 + low;
    MergeSort(result, low, mid);
    MergeSort(result, mid + 1, high);
    Merge(result, low, mid, high);
  }
}

void SortDetectionResult(DetectionResult* result) {
  size_t low = 0;
  size_t high = result->scores.size();
  if (high == 0) {
    return;
  }
  high = high - 1;
  MergeSort(result, low, high);
}

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
