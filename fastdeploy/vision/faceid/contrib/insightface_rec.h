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

namespace fastdeploy {

namespace vision {

namespace faceid {

class FASTDEPLOY_DECL InsightFaceRecognitionModel : public FastDeployModel {
 public:
  InsightFaceRecognitionModel(
      const std::string& model_file, const std::string& params_file = "",
      const RuntimeOption& custom_option = RuntimeOption(),
      const ModelFormat& model_format = ModelFormat::ONNX);

  virtual std::string ModelName() const { return "deepinsight/insightface"; }

  // tuple of (width, height), default (112, 112)
  std::vector<int> size;
  std::vector<float> alpha;
  std::vector<float> beta;
  // whether to swap the B and R channel, such as BGR->RGB, default true.
  bool swap_rb;
  // whether to apply l2 normalize to embedding values, default;
  bool l2_normalize;

  virtual bool Predict(cv::Mat* im, FaceRecognitionResult* result);

  virtual bool Initialize();

  virtual bool Preprocess(Mat* mat, FDTensor* output);

  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           FaceRecognitionResult* result);

  // RKNPU2 can run normalize and hwc2chw on the NPU.
  // This function is used to close normalize and hwc2chw operations in preprocessing.
  void DisableNormalizeAndPermute();

  private:
  // for recording the switch of normalize and hwc2chw
  bool switch_of_nor_and_per = true;  
};

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
