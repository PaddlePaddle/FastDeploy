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

static void RemoveSmallConnectedArea(cv::Mat* alpha_pred,
                                     float threshold = 0.05f) {
  // 移除小的联通区域和噪点 开闭合形态学处理
  // 假设输入的是透明度alpha, 值域(0.,1.)
  cv::Mat gray, binary;
  (*alpha_pred).convertTo(gray, CV_8UC1, 255.f);
  // 255 * 0.05 ~ 13
  unsigned int binary_threshold = static_cast<unsigned int>(255.f * threshold);
  cv::threshold(gray, binary, binary_threshold, 255, cv::THRESH_BINARY);
  // morphologyEx with OPEN operation to remove noise first.
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3),
                                          cv::Point(-1, -1));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  // Computationally connected domain
  cv::Mat labels = cv::Mat::zeros((*alpha_pred).size(), CV_32S);
  cv::Mat stats, centroids;
  int num_labels =
      cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
  if (num_labels <= 1) {
    // no noise, skip.
    return;
  }
  // find max connected area, 0 is background
  int max_connected_id = 1;  // 1,2,...
  int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
  for (int i = 1; i < num_labels; ++i) {
    int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (tmp_connected_area > max_connected_area) {
      max_connected_area = tmp_connected_area;
      max_connected_id = i;
    }
  }
  const int h = (*alpha_pred).rows;
  const int w = (*alpha_pred).cols;
  // remove small connected area.
  for (int i = 0; i < h; ++i) {
    int* label_row_ptr = labels.ptr<int>(i);
    float* alpha_row_ptr = (*alpha_pred).ptr<float>(i);
    for (int j = 0; j < w; ++j) {
      if (label_row_ptr[j] != max_connected_id) alpha_row_ptr[j] = 0.f;
    }
  }
}

cv::Mat Visualize::VisMattingAlpha(const cv::Mat& im,
                                   const MattingResult& result,
                                   const cv::Mat& background,
                                   bool remove_small_connected_area) {
  // 只可视化alpha，fgr(前景)本身就是一张图 不需要可视化
  FDASSERT((!im.empty()), "im can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels mat!");
  if (!background.empty()) {
    FDASSERT((background.channels() == 3),
             "Only support 3 channels background mat!");
  }
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
  if (background.empty()) {
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
  } else {
    int bg_height = background.rows;
    int bg_width = background.cols;
    cv::Mat background_copy = background.clone();
    if ((background_copy).type() != CV_8UC3) {
      (background_copy).convertTo((background_copy), CV_8UC3);
    }
    if ((bg_height != height) || (bg_width != width)) {
      cv::resize(background, background_copy, cv::Size(width, height));
    }
    uchar* background_data = static_cast<uchar*>(background_copy.data);
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        float alpha_val = alpha_data[i * width + j];
        for (size_t c = 0; c < 3; ++c) {
          vis_data[i * width * 3 + j * 3 + c] = cv::saturate_cast<uchar>(
              static_cast<float>(im_data[i * width * 3 + j * 3 + c]) *
                  alpha_val +
              (1.f - alpha_val) * background_data[i * width * 3 + j * 3 + c]);
        }
      }
    }
  }

  return vis_img;
}

// cv::Mat Visualize::VisMattingAlphaWithBackground(const cv::Mat& im,
//                                    const cv::Mat& background,
//                                    const MattingResult& result,
//                                    bool remove_small_connected_area) {
//   // 只可视化alpha，fgr(前景)本身就是一张图 不需要可视化
//   FDASSERT((!im.empty()), "im can't be empty!");
//   FDASSERT((im.channels() == 3), "im only support 3 channels mat!");
//   FDASSERT((!im.empty()), "background image can't be empty!");
//   FDASSERT((im.channels() == 3), "background image only support 3 channels
//   mat!");
//   auto vis_img = im.clone();
//   int out_h = static_cast<int>(result.shape[0]);
//   int out_w = static_cast<int>(result.shape[1]);
//   int height = im.rows;
//   int width = im.cols;
//   int bg_height = background.rows;
//   int bg_width = background.cols;
//   // alpha to cv::Mat && 避免resize等操作修改外部数据
//   std::vector<float> alpha_copy;
//   alpha_copy.assign(result.alpha.begin(), result.alpha.end());
//   float* alpha_ptr = static_cast<float*>(alpha_copy.data());
//   cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);
//   if (remove_small_connected_area) {
//     RemoveSmallConnectedArea(&alpha, 0.05f);
//   }
//   if ((out_h != height) || (out_w != width)) {
//     cv::resize(alpha, alpha, cv::Size(width, height));
//   }
//   if ((bg_height != height) || (bg_width != width)) {
//     cv::resize(background, background, cv::Size(width, height));
//   }

//   if ((vis_img).type() != CV_8UC3) {
//     (vis_img).convertTo((vis_img), CV_8UC3);
//   }

//   uchar* vis_data = static_cast<uchar*>(vis_img.data);
//   uchar* im_data = static_cast<uchar*>(im.data);
//   uchar* background_data = static_cast<uchar*>(background.data);
//   float* alpha_data = reinterpret_cast<float*>(alpha.data);
//   for (size_t i = 0; i < height; ++i) {
//     for (size_t j = 0; j < width; ++j) {
//       float alpha_val = alpha_data[i * width + j];
//       for (size_t c=0; c<3;++c){
//         vis_data[i * width * 3 + j * 3 + c] = cv::saturate_cast<uchar>(
//           static_cast<float>(im_data[i * width * 3 + j * 3 + c]) * alpha_val
//           +
//           (1.f - alpha_val) * background_data[i * width * 3 + j * 3 + c]);
//       }
//     }
//   }
//   return vis_img;
// }

}  // namespace vision
}  // namespace fastdeploy
#endif
