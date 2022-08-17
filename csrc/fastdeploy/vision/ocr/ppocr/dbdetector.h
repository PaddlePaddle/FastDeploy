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
#include "fastdeploy/vision/ppocr/utils/ocr_postprocess_op.h"

namespace fastdeploy {
namespace vision {
namespace ppocr {

class FASTDEPLOY_DECL DBDetector : public FastDeployModel {
 public:
  // 当model_format为ONNX时，无需指定params_file
  // 当model_format为Paddle时，则需同时指定model_file & params_file
  DBDetector(const std::string& model_file, const std::string& params_file = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);

  // 定义模型的名称
  std::string ModelName() const { return "ppocr/ocr_det"; }

  // 模型预测接口，即用户调用的接口
  virtual bool Predict(cv::Mat* im,
                       std::vector<std::vector<std::vector<int>>>& boxes);

  // Copy from ppocr2.5-ocr_det.h
  // pre&post process parameters
  int max_side_len;

  float ratio_h{};
  float ratio_w{};

  double det_db_thresh;
  double det_db_box_thresh;
  double det_db_unclip_ratio;
  std::string det_db_score_mode;
  bool use_dilation;

  std::vector<float> mean;
  std::vector<float> scale;
  bool is_scale;

 private:
  // 初始化函数，包括初始化后端，以及其它模型推理需要涉及的操作
  bool Initialize();

  // 输入图像预处理操作
  // 由于预处理函数问题，暂时用cv::Mat 替代自定义的Mat数据结构
  // FDTensor为预处理后的Tensor数据，传给后端进行推理
  // im_info为预处理过程保存的数据，在后处理中需要用到
  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  // 后端推理结果后处理，输出给用户
  // infer_result 为后端推理后的输出Tensor
  // result 为模型预测的结果
  // im_info 为预处理记录的信息，后处理用于还原box
  bool Postprocess(FDTensor& infer_result,
                   std::vector<std::vector<std::vector<int>>>* boxes,
                   const std::map<std::string, std::array<float, 2>>& im_info);

  // OCR后处理类
  PostProcessor post_processor_;
};

}  // namespace ppocr
}  // namespace vision
}  // namespace fastdeploy
