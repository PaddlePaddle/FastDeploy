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
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy{

namespace vision{

namespace facedet{

class FASTDEPLOY_DECL Yolov7FacePostprocessor{
  public:
  Yolov7FacePostprocessor();

  bool Run(const std::vector<FDTensor>& infer_result,
           std::vector<FaceDetectionResult>* results,
           const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info);

  protected:
  float conf_threshold_;
  float nms_threshold_;
  bool multi_label_;
  float max_wh_;
};

}//facedet

}//vision

}//fastdeploy