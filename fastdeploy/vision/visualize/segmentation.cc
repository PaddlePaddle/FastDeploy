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

cv::Mat VisSegmentation(const cv::Mat& im, const SegmentationResult& result,
                        float weight) {
  auto color_map = GenerateColorMap(1000);
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  int64_t index = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int category_id = result.label_map[index++];
      vis_img.at<cv::Vec3b>(i, j)[0] = color_map[3 * category_id + 0];
      vis_img.at<cv::Vec3b>(i, j)[1] = color_map[3 * category_id + 1];
      vis_img.at<cv::Vec3b>(i, j)[2] = color_map[3 * category_id + 2];
    }
  }
  cv::addWeighted(im, 1.0 - weight, vis_img, weight, 0, vis_img);
  return vis_img;
}

cv::Mat Visualize::VisSegmentation(const cv::Mat& im,
                                   const SegmentationResult& result) {
  FDWARNING << "DEPRECATED: fastdeploy::vision::Visualize::VisSegmentation is "
               "deprecated, please use fastdeploy::vision:VisSegmentation "
               "function instead."
            << std::endl;
  auto color_map = GetColorMap();
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  int64_t index = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int category_id = result.label_map[index++];
      vis_img.at<cv::Vec3b>(i, j)[0] = color_map[3 * category_id + 0];
      vis_img.at<cv::Vec3b>(i, j)[1] = color_map[3 * category_id + 1];
      vis_img.at<cv::Vec3b>(i, j)[2] = color_map[3 * category_id + 2];
    }
  }
  cv::addWeighted(im, .5, vis_img, .5, 0, vis_img);
  return vis_img;
}

}  // namespace vision
}  // namespace fastdeploy
#endif
