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
  if (result.contain_masks) {
    FDASSERT(result.boxes.size() == result.masks.size(),
             "The size of masks must be same as the size of boxes!");
  }
  auto color_map = GetColorMap();
  int h = im.rows;
  int w = im.cols;
  auto vis_im = im.clone();
  for (size_t i = 0; i < result.boxes.size(); ++i) {
    if (result.scores[i] < score_threshold) {
      continue;
    }
    int x1 = static_cast<int>(result.boxes[i][0]);
    int y1 = static_cast<int>(result.boxes[i][1]);
    int x2 = static_cast<int>(result.boxes[i][2]);
    int y2 = static_cast<int>(result.boxes[i][3]);
    int box_h = y2 - y1;
    int box_w = x2 - x2;
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
    origin.x = x1;
    origin.y = y1;
    cv::Rect rect(x1, y1, box_w, box_h);
    cv::rectangle(vis_im, rect, rect_color, line_size);
    cv::putText(vis_im, text, origin, font, font_size,
                cv::Scalar(255, 255, 255), 1);
    // draw instance mask
    if (result.contain_masks) {
      int mask_h = static_cast<int>(result.masks[i].shape[0]);
      int mask_w = static_cast<int>(result.masks[i].shape[1]);
      // non-const pointer for cv:Mat constructor.
      int32_t* mask_raw_data = const_cast<int32_t*>(
          static_cast<const int32_t*>(result.masks[i].Data()));
      cv::Mat mask(mask_h, mask_w, CV_32SC1, mask_raw_data);  // ref only
      if ((mask_h != box_h) || (mask_w != box_w)) {
        cv::resize(mask, mask, cv::Size(box_w, box_h));
      }
      cv::Mat box_im = vis_im(rect).clone();  // allocate continuous memory
      cv::Mat mask_im(mask_h, mask_w, CV_8UC3, rect_color);
      uchar* box_im_data = static_cast<uchar*>(box_im.data);
      uchar* mask_im_data = static_cast<uchar*>(mask_im.data);
      int32_t* mask_data = reinterpret_cast<int32_t*>(mask.data);
      for (size_t i = 0; i < mask_h; ++i) {
        for (size_t j = 0; j < mask_w; ++j) {
          // add mask color values
          int32_t mask_value = mask_data[i * mask_w + j];
          for (size_t c = 0; c < 3; ++c) {
            uchar mask_im_value = mask_im_data[i * mask_w * 3 + j * 3 + c];
            uchar box_im_value = box_im_data[i * mask_w * 3 + j * 3 + c];
            if (mask_value == 0) {
              mask_im_data[i * mask_w * 3 + j * 3 + c] = box_im_value;
            } else {
              mask_im_data[i * mask_w * 3 + j * 3 + c] =
                  cv::saturate_cast<uchar>(
                      static_cast<float>(mask_im_value) * 0.5f +
                      static_cast<float>(box_im_value) * 0.5f);
            }
          }
        }
      }
      // replace mask
      mask_im.copyTo(vis_im(rect));
    }
  }
  return vis_im;
}

}  // namespace vision
}  // namespace fastdeploy
#endif
