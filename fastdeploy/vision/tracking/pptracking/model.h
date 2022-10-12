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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/tracking/pptracking/tracker.h"

namespace fastdeploy {
namespace vision {
namespace tracking {

class LetterBoxResize:  public Processor{
public:
  LetterBoxResize(const std::vector<int>& target_size, const std::vector<float>& color){
    target_size_=target_size;
    color_=color;
  }
  bool ImplByOpenCV(Mat* mat) override{

    if (mat->Channels() != color_.size()) {
      FDERROR << "Pad: Require input channels equals to size of padding value, "
                 "but now channels = "
              << mat->Channels()
              << ", the size of padding values = " << color_.size() << "."
              << std::endl;
      return false;
    }
    // generate scale_factor
    int origin_w = mat->Width();
    int origin_h = mat->Height();
    int target_h = target_size_[0];
    int target_w = target_size_[1];
    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);

    int new_shape_w = std::round(mat->Width() * resize_scale);
    int new_shape_h = std::round(mat->Height() * resize_scale);
    float padw = (target_size_[1] - new_shape_w) / 2.;
    float padh = (target_size_[0] - new_shape_h) / 2.;
    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    Resize::Run(mat,new_shape_w,new_shape_h);
    Pad::Run(mat,top,bottom,left,right,color_);
    return true;
  }
  std::string Name() override { return "LetterBoxResize"; }

private:
  std::vector<int> target_size_;
  std::vector<float> color_;
};

class FASTDEPLOY_DECL PPTracking: public FastDeployModel {

public:
  PPTracking(const std::string& model_file,
             const std::string& params_file,
             const std::string& config_file,
             const RuntimeOption& custom_option = RuntimeOption(),
             const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const override { return "pptracking"; }

  virtual bool Predict(cv::Mat* img, MOTResult* result);



private:

  bool BuildPreprocessPipelineFromConfig();
  bool Initialize();
  void GetNmsInfo();

//  void LetterBoxResize(Mat* im);

  bool Preprocess(Mat* img, std::vector<FDTensor>* outputs);

  bool Postprocess(std::vector<FDTensor>& infer_result, MOTResult *result);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  float draw_threshold_;
  float conf_thresh_;
  float tracked_thresh_;
  float min_box_area_;
  bool is_scale_ = true;

  // configuration for nms
  int64_t background_label = -1;
  int64_t keep_top_k = 300;
  float nms_eta = 1.0;
  float nms_threshold = 0.7;
  float score_threshold = 0.01;
  int64_t nms_top_k = 10000;
  bool normalized = true;
  bool has_nms_ = true;

};

} // namespace tracking
} // namespace vision
} // namespace fastdeploy

