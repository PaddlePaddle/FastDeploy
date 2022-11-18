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
#include "fastdeploy/vision/facedet/yolov7-face/preprocessor.h"
#include "fastdeploy/vision/facedet/yolov7-face/postprocessor.h"

namespace fastdeploy{

namespace vision{

namespace facedet{

class FASTDEPLOY_DECL YOLOv7Face: public FastDeployModel{
  public:
  YOLOv7Face(const std::string& model_file, const std::string& params_file = "",
              const RuntimeOption& custom_option = RuntimeOption(),
              const ModelFormat& model_format = ModelFormat::ONNX);
  
  std::string ModelName(){return "yolov7-face";}

  virtual bool Predict(cv::Mat* im, FaceDetectionResult* result);

  virtual bool Predict(const cv::Mat& im, FaceDetectionResult* result);

  virtual bool BatchPredict(const std::vector<cv::Mat>& images,
                            std::vector<FaceDetectionResult>* results);

  protected:
  bool Initialize();
  Yolov7FacePreprocessor preprocessor_;
  Yolov7FacePostprocessor postprocessor_;
    
  };

}//namespace facedet

}//namespace vision

}//namespace fastdeploy