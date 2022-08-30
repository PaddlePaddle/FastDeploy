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
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace detection {

cv::Point2f get_3rd_point(cv::Point2f& a, cv::Point2f& b);

std::vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad);

void get_affine_transform(std::vector<float>& center, std::vector<float>& scale,
                          float rot, std::vector<int>& output_size,
                          cv::Mat& trans, int inv);
void affine_tranform(float pt_x, float pt_y, cv::Mat& trans,
                     std::vector<float>& preds, int p);

void transform_preds(std::vector<float>& coords, std::vector<float>& center,
                     std::vector<float>& scale, std::vector<int>& output_size,
                     std::vector<int>& dim, std::vector<float>& target_coords);

void get_final_preds(std::vector<float>& heatmap, std::vector<int>& dim,
                     std::vector<int64_t>& idxout, std::vector<int>& idxdim,
                     std::vector<float>& center, std::vector<float> scale,
                     std::vector<float>& preds, bool DARK);

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
