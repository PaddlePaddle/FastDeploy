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
             const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "ppocr/ocr_det"; }

  virtual bool Predict(cv::Mat* im,
                       std::vector<std::array<int, 8>>* boxes_result);

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
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result,
                   std::vector<std::array<int, 8>>* boxes_result,
                   const std::map<std::string, std::array<float, 2>>& im_info);

  PostProcessor post_processor_;
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
