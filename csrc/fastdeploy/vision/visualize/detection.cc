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
#include "opencv2/imgproc/imgproc.hpp"

namespace fastdeploy {
namespace vision {

// Default only support visualize num_classes <= 1000
// If need to visualize num_classes > 1000
// Please call Visualize::GetColorMap(num_classes) first
cv::Mat Visualize::VisDetection(const cv::Mat& im,
                                const DetectionResult& result,
                                float score_threshold, int line_size,
                                float font_size) {
  auto color_map = GetColorMap();
  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    cv::Rect rect(result.boxes[i][0], result.boxes[i][1],
                  result.boxes[i][2] - result.boxes[i][0],
                  result.boxes[i][3] - result.boxes[i][1]);
    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
    std::string id = std::to_string(result.label_ids[i]);
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = id + "," + score;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
    cv::Point origin;
    origin.x = rect.x;
    origin.y = rect.y;
    cv::Rect text_background =
        cv::Rect(result.boxes[i][0], result.boxes[i][1] - text_size.height,
                 text_size.width, text_size.height);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
  }
  return vis_im;
}

}  // namespace vision
}  // namespace fastdeploy
#endif
