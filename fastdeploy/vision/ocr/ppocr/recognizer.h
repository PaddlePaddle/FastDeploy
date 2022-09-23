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

class FASTDEPLOY_DECL Recognizer : public FastDeployModel {
 public:
  Recognizer();
  Recognizer(const std::string& model_file, const std::string& params_file = "",
             const std::string& label_path = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "ppocr/ocr_rec"; }

  virtual bool Predict(cv::Mat* img,
                       std::tuple<std::string, float>* rec_result);

  // pre & post parameters
  std::vector<std::string> label_list;
  int rec_batch_num;
  int rec_img_h;
  int rec_img_w;
  std::vector<int> rec_image_shape;

  std::vector<float> mean;
  std::vector<float> scale;
  bool is_scale;

 private:
  bool Initialize();

  bool Preprocess(Mat* img, FDTensor* outputs,
                  const std::vector<int>& rec_image_shape);

  bool Postprocess(FDTensor& infer_result,
                   std::tuple<std::string, float>* rec_result);
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
