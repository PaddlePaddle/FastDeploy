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
  // 当model_format为ONNX时，无需指定params_file
  // 当model_format为Paddle时，则需同时指定model_file & params_file
  Recognizer(const std::string& model_file, const std::string& params_file = "",
             const std::string& label_path = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);

  // 定义模型的名称
  std::string ModelName() const { return "ppocr/ocr_rec"; }

  // 模型预测接口，即用户调用的接口
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
  // 初始化函数，包括初始化后端，以及其它模型推理需要涉及的操作
  bool Initialize();

  // 输入图像预处理操作
  bool Preprocess(Mat* img, FDTensor* outputs,
                  const std::vector<int>& rec_image_shape);

  // 后端推理结果后处理，输出给用户
  // infer_result 为后端推理后的输出Tensor
  bool Postprocess(FDTensor& infer_result,
                   std::tuple<std::string, float>* rec_result);
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
