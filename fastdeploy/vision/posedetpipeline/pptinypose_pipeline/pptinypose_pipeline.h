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

#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/detection/ppdet/model.h"
#include "fastdeploy/vision/keypointdet/pptinypose/pptinypose.h"

namespace fastdeploy {
namespace application {
namespace posedetpipeline {

class FASTDEPLOY_DECL PPTinyPosePipeline : public FastDeployModel {
 public:
  PPTinyPosePipeline(
      fastdeploy::vision::detection::PPYOLOE* det_model,
      fastdeploy::vision::keypointdetection::PPTinyPose* pptinypose_model);

  virtual bool Predict(cv::Mat* img,
                       fastdeploy::vision::KeyPointDetectionResult* result);

 protected:
  fastdeploy::vision::detection::PPYOLOE* detector_ = nullptr;
  fastdeploy::vision::keypointdetection::PPTinyPose* pptinypose_model_ =
      nullptr;

  virtual bool Detect(cv::Mat* img,
                      fastdeploy::vision::DetectionResult* result);
  virtual bool KeypointDetect(
      cv::Mat* img, fastdeploy::vision::KeyPointDetectionResult* result,
      fastdeploy::vision::DetectionResult& detection_result);
};

}  // namespace posedetpipeline
}  // namespace application
}  // namespace fastdeploy
