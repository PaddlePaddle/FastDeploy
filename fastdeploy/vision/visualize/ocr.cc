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

#include "fastdeploy/vision/visualize/visualize.h"

namespace fastdeploy {
namespace vision {

cv::Mat VisOcr(const cv::Mat& im, const OCRResult& ocr_result,
               const float score_threshold) {
  auto vis_im = im.clone();

  for (int n = 0; n < ocr_result.boxes.size(); n++) {
    if (ocr_result.rec_scores[n] < score_threshold) {
      continue;
    }
    cv::Point rook_points[4];

    for (int m = 0; m < 4; m++) {
      rook_points[m] = cv::Point(int(ocr_result.boxes[n][m * 2]),
                                 int(ocr_result.boxes[n][m * 2 + 1]));
    }

    const cv::Point* ppt[1] = {rook_points};
    int npt[] = {4};
    cv::polylines(vis_im, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  return vis_im;
}

cv::Mat Visualize::VisOcr(const cv::Mat& im, const OCRResult& ocr_result) {
  FDWARNING
      << "DEPRECATED: fastdeploy::vision::Visualize::VisOcr is deprecated, "
         "please use fastdeploy::vision:VisOcr function instead."
      << std::endl;
  auto vis_im = im.clone();

  for (int n = 0; n < ocr_result.boxes.size(); n++) {
    cv::Point rook_points[4];

    for (int m = 0; m < 4; m++) {
      rook_points[m] = cv::Point(int(ocr_result.boxes[n][m * 2]),
                                 int(ocr_result.boxes[n][m * 2 + 1]));
    }

    const cv::Point* ppt[1] = {rook_points};
    int npt[] = {4};
    cv::polylines(vis_im, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  return vis_im;
}

}  // namespace vision
}  // namespace fastdeploy
