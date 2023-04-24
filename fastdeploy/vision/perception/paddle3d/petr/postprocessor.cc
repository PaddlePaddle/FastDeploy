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

#include "fastdeploy/vision/perception/paddle3d/petr/postprocessor.h"

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace perception {

PetrPostprocessor::PetrPostprocessor() {}

bool PetrPostprocessor::Run(const std::vector<FDTensor>& tensors,
                             std::vector<PerceptionResult>* results) {
  results->resize(1);
  (*results)[0].Clear();
  (*results)[0].Reserve(tensors[0].shape[0]);
  if (tensors[0].dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  const float* data = reinterpret_cast<const float*>(tensors[0].Data());
  auto result = &(*results)[0];
  for (int i = 0; i < tensors[0].shape[0] * tensors[0].shape[1]; i += 9) {
    // item 1 ~ 3   :  box3d w, h, l
    // item 4 ~ 6   :  box3d bottom center x, y, z
    // item 7       :  box3d yaw angle
    // item 8 ~ 9   :  speed x,y
    // item 10      :  labels
    // item 11      :  score
    std::vector<float> vec(data + i, data + i + 11);
    result->scores.push_back(vec[10]);
    result->label_ids.push_back(vec[9]);
    result->boxes.emplace_back(std::array<float, 7>{
        0, 0, 0, 0, vec[0], vec[1], vec[2]});
    result->center.emplace_back(std::array<float, 3>{vec[3], vec[4], vec[5]});
    result->yaw_angle.push_back(vec[6]);
    result->velocity.push_back(std::array<float, 3>{vec[7], vec[8]});
  }
  return true;
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
