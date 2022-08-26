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

#ifdef ENABLE_VISION_VISUALIZE

#include "fastdeploy/vision/visualize/visualize.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace fastdeploy {
namespace vision {

cv::Mat Visualize::VisMattingAlpha(const cv::Mat& im,
                                   const MattingResult& result,
                                   bool remove_small_connected_area) {
  // 只可视化alpha，fgr(前景)本身就是一张图 不需要可视化
  FDASSERT((!im.empty()), "im can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels mat!");

  auto vis_img = im.clone();
  int out_h = static_cast<int>(result.shape[0]);
  int out_w = static_cast<int>(result.shape[1]);
  int height = im.rows;
  int width = im.cols;
  // alpha to cv::Mat && 避免resize等操作修改外部数据
  std::vector<float> alpha_copy;
  alpha_copy.assign(result.alpha.begin(), result.alpha.end());
  float* alpha_ptr = static_cast<float*>(alpha_copy.data());
  cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);
  if (remove_small_connected_area) {
    RemoveSmallConnectedArea(&alpha, 0.05f);
  }
  if ((out_h != height) || (out_w != width)) {
    cv::resize(alpha, alpha, cv::Size(width, height));
  }

  if ((vis_img).type() != CV_8UC3) {
    (vis_img).convertTo((vis_img), CV_8UC3);
  }

  uchar* vis_data = static_cast<uchar*>(vis_img.data);
  uchar* im_data = static_cast<uchar*>(im.data);
  float* alpha_data = reinterpret_cast<float*>(alpha.data);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      float alpha_val = alpha_data[i * width + j];
      vis_data[i * width * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 0]) * alpha_val +
          (1.f - alpha_val) * 153.f);
      vis_data[i * width * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 1]) * alpha_val +
          (1.f - alpha_val) * 255.f);
      vis_data[i * width * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * width * 3 + j * 3 + 2]) * alpha_val +
          (1.f - alpha_val) * 120.f);
    }
  }
  return vis_img;
}

}  // namespace vision
}  // namespace fastdeploy
#endif
