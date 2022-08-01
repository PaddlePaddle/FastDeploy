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

#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/visualize/visualize.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace fastdeploy {
namespace vision {

void Visualize::VisSegmentation(const cv::Mat& im,
                                const SegmentationResult& result,
                                cv::Mat* vis_img, const int& num_classes) {
  int origin_h = im.rows;
  int origin_w = im.cols;
  auto color_map = GetColorMap(num_classes);
  int mask_h = result.masks.size();
  int mask_w = result.masks[0].size();
  *vis_img = cv::Mat::zeros(origin_h, origin_w, CV_8UC3);
  cv::Mat mask_mat(mask_h, mask_w, CV_32FC1);

  for (int i = 0; i < mask_h; ++i) {
    for (int j = 0; j < mask_w; ++j) {
      mask_mat.at<float>(i, j) = static_cast<float>(result.masks[i][j]);
    }
  }
  Mat mat(mask_mat);
  if (origin_h != mask_h || origin_w != mask_w) {
    Resize::Run(&mat, origin_w, origin_h);
  }
#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat* im_mask = mat.GetGpuMat();
#else
  cv::Mat* im_mask = mat.GetCpuMat();
#endif

  int64_t index = 0;
  for (int i = 0; i < origin_h; i++) {
    for (int j = 0; j < origin_w; j++) {
      int category_id = static_cast<int>((*im_mask).at<float>(i, j));
      vis_img->at<cv::Vec3b>(i, j)[0] = color_map[3 * category_id + 0];
      vis_img->at<cv::Vec3b>(i, j)[1] = color_map[3 * category_id + 1];
      vis_img->at<cv::Vec3b>(i, j)[2] = color_map[3 * category_id + 2];
    }
  }
  cv::addWeighted(im, .5, *vis_img, .5, 0, *vis_img);
}

}  // namespace vision
}  // namespace fastdeploy
#endif
