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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

#include "fastdeploy/vision/detection/ppdet/pptinypose_utils.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace detection {

class FASTDEPLOY_DECL PPTinyPose : public FastDeployModel {
 public:
  PPTinyPose(const std::string& model_file, const std::string& params_file,
             const std::string& config_file,
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "PaddleDetection/PPTinyPose"; }

  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);

  bool Postprocess(std::vector<FDTensor>& infer_result,
                   KeyPointDetectionResult* result,
                   const std::vector<float>& center,
                   const std::vector<float>& scale);

  bool Predict(cv::Mat* im, KeyPointDetectionResult* result,
               DetectionResult* detection_result = nullptr);
  bool use_dark = true;

 private:
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  float threshold;
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
