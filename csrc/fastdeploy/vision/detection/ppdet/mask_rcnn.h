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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace detection {

class FASTDEPLOY_DECL MaskRCNN : public FastDeployModel {
 public:
  MaskRCNN(const std::string& model_file, const std::string& params_file,
           const std::string& config_file,
           const RuntimeOption& custom_option = RuntimeOption(),
           const Frontend& model_format = Frontend::PADDLE);

  virtual std::string ModelName() const { return "PaddleDetection/MaskRCNN"; }

  virtual bool Initialize();

  virtual bool BuildPreprocessPipelineFromConfig();

  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* inputs);

  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result);

  virtual bool Predict(cv::Mat* im, DetectionResult* result);

 protected:
  MaskRCNN() {}

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;

  std::vector<float> GetScaleFactorData(const Mat& mat, int origin_h,
                                        int origin_w);

  std::vector<float> GetImShapeData(const Mat& mat);
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
