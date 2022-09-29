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
#pragma once

#include "fastdeploy/vision/common/result.h"
#include "opencv2/imgproc/imgproc.hpp"
namespace fastdeploy {
namespace vision {

// This class will deprecated, please not use it
class FASTDEPLOY_DECL Visualize {
 public:
  static int num_classes_;
  static std::vector<int> color_map_;
  static const std::vector<int>& GetColorMap(int num_classes = 1000);
  static cv::Mat VisDetection(const cv::Mat& im, const DetectionResult& result,
                              float score_threshold = 0.0, int line_size = 1,
                              float font_size = 0.5f);
  static cv::Mat VisFaceDetection(const cv::Mat& im,
                                  const FaceDetectionResult& result,
                                  int line_size = 1, float font_size = 0.5f);
  static cv::Mat VisSegmentation(const cv::Mat& im,
                                 const SegmentationResult& result);
  static cv::Mat VisMattingAlpha(const cv::Mat& im, const MattingResult& result,
                                 bool remove_small_connected_area = false);
  static cv::Mat RemoveSmallConnectedArea(const cv::Mat& alpha_pred,
                                          float threshold);
  static cv::Mat SwapBackgroundMatting(
      const cv::Mat& im, const cv::Mat& background, const MattingResult& result,
      bool remove_small_connected_area = false);
  static cv::Mat SwapBackgroundSegmentation(const cv::Mat& im,
                                            const cv::Mat& background,
                                            int background_label,
                                            const SegmentationResult& result);
  static cv::Mat VisOcr(const cv::Mat& srcimg, const OCRResult& ocr_result);
};

std::vector<int> GenerateColorMap(int num_classes = 1000);
cv::Mat RemoveSmallConnectedArea(const cv::Mat& alpha_pred, float threshold);
FASTDEPLOY_DECL cv::Mat VisDetection(const cv::Mat& im,
                                     const DetectionResult& result,
                                     float score_threshold = 0.0,
                                     int line_size = 1, float font_size = 0.5f);
FASTDEPLOY_DECL cv::Mat VisFaceDetection(const cv::Mat& im,
                                         const FaceDetectionResult& result,
                                         int line_size = 1,
                                         float font_size = 0.5f);
FASTDEPLOY_DECL cv::Mat VisSegmentation(const cv::Mat& im,
                                        const SegmentationResult& result,
                                        float weight = 0.5);
FASTDEPLOY_DECL cv::Mat VisMatting(const cv::Mat& im,
                                   const MattingResult& result,
                                   bool remove_small_connected_area = false);
FASTDEPLOY_DECL cv::Mat VisOcr(const cv::Mat& im, const OCRResult& ocr_result);
FASTDEPLOY_DECL cv::Mat SwapBackground(
    const cv::Mat& im, const cv::Mat& background, const MattingResult& result,
    bool remove_small_connected_area = false);
FASTDEPLOY_DECL cv::Mat SwapBackground(const cv::Mat& im,
                                       const cv::Mat& background,
                                       const SegmentationResult& result,
                                       int background_label);

}  // namespace vision
}  // namespace fastdeploy
#endif
