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

namespace fastdeploy {
namespace vision {

cv::Mat Visualize::VisKeypointDetection(const cv::Mat& im,
                                        const KeyPointDetectionResult& results,
                                        float conf_threshold) {
  const int edge[][2] = {{0, 1},   {0, 2},  {1, 3},   {2, 4},   {3, 5},
                         {4, 6},   {5, 7},  {6, 8},   {7, 9},   {8, 10},
                         {5, 11},  {6, 12}, {11, 13}, {12, 14}, {13, 15},
                         {14, 16}, {11, 12}};
  auto colormap = GetColorMap();
  cv::Mat vis_img = im.clone();
  for (int i = 0; i < results.num_joints; i++) {
    if (results.keypoints[i * 3] > conf_threshold) {
      int x_coord = int(results.keypoints[i * 3 + 1]);
      int y_coord = int(results.keypoints[i * 3 + 2]);
      cv::circle(vis_img, cv::Point2d(x_coord, y_coord), 1,
                 cv::Scalar(0, 0, 255), 2);
    }
  }
  for (int i = 0; i < results.num_joints; i++) {
    int x_start = int(results.keypoints[edge[i][0] * 3 + 1]);
    int y_start = int(results.keypoints[edge[i][0] * 3 + 2]);
    int x_end = int(results.keypoints[edge[i][1] * 3 + 1]);
    int y_end = int(results.keypoints[edge[i][1] * 3 + 2]);
    cv::line(vis_img, cv::Point2d(x_start, y_start), cv::Point2d(x_end, y_end),
             colormap[i], 1);
  }
  return vis_img;
}

}  // namespace vision
}  // namespace fastdeploy
#endif
