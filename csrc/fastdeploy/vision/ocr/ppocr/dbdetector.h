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

class FASTDEPLOY_DECL DBDetector : public FastDeployModel {
 public:
  DBDetector();

  DBDetector(const std::string& model_file, const std::string& params_file = "",
             const RuntimeOption& custom_option = RuntimeOption(),
             const Frontend& model_format = Frontend::PADDLE);

  // 定义模型的名称
  std::string ModelName() const { return "ppocr/ocr_det"; }

  // 模型预测接口，即用户调用的接口
  virtual bool Predict(cv::Mat* im,
                       std::vector<std::vector<std::vector<int>>>* boxes);

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

  // FDTensor为预处理后的Tensor数据，传给后端进行推理
  // im_info为预处理过程保存的数据，在后处理中需要用到
  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  // 后端推理结果后处理，输出给用户
  bool Postprocess(FDTensor& infer_result,
                   std::vector<std::vector<std::vector<int>>>* boxes,
                   const std::map<std::string, std::array<float, 2>>& im_info);

  // OCR后处理类
  PostProcessor post_processor_;
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
