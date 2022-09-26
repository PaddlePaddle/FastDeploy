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
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

class FASTDEPLOY_DECL Classifier : public FastDeployModel {
 public:
  Classifier();
  Classifier(const std::string& model_file, const std::string& params_file = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "ppocr/ocr_cls"; }

  virtual bool Predict(cv::Mat* img, std::tuple<int, float>* result);

  // pre & post parameters
  float cls_thresh;
  std::vector<int> cls_image_shape;
  int cls_batch_num;

  std::vector<float> mean;
  std::vector<float> scale;
  bool is_scale;

 private:
  bool Initialize();

  bool Preprocess(Mat* img, FDTensor* output);

  bool Postprocess(FDTensor& infer_result, std::tuple<int, float>* result);
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
